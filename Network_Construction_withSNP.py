import pandas as pd
import numpy as np
from cobra.io import load_json_model
from graph import Graph
import time
import json
import re
from collections import defaultdict, deque
import sys
sys.path.append(r'../MetabOmics_Aycan/omicNetwork')

# Load the synonym mapping for recon3D (gene_name -> [uniprot_ids])
with open('/home/cakmaklab/Documents/MetabOmics_Aycan/omicNetwork/Databases/recon3D_uniprot_mapping.json', 'r') as json_file:
    recon3D_uniprot_mapping = json.load(json_file)

# Load the new synonym mapping (gene_name -> [uniprot_ids])
with open('/home/cakmaklab/Documents/MetabOmics_Aycan/omicNetwork/Databases/uniprot_synonym_mapping.json', 'r') as json_file:
    synonym_mapping_dict = json.load(json_file)

'''def select_uniprot_id(uniprot_ids):
    for prefix in ['P', 'Q', 'O']:
        for uid in uniprot_ids:
            if uid.startswith(prefix):
                return uid
    return uniprot_ids[0]  # fallback'''

def parse_gpr_expression(gpr_expression, uniprot_synonym_mapping):
    # Split OR conditions
    or_conditions = re.split(r' or ', gpr_expression)

    gene_lists = []
    for condition in or_conditions:
        genes = condition.strip('()').split(" and ")
        temp = []
        for gene in genes:
            gene = gene.replace('(', '').replace(')', '').strip()
            # Use gene.name as key in uniprot_synonym_mapping
            if gene in uniprot_synonym_mapping:
                uniprot_id = uniprot_synonym_mapping[gene]
                #selected_uniprot = select_uniprot_id(uniprot_ids)
                temp.append(uniprot_id)
            else:
                # Gene not found in mapping — optionally log or handle
                pass
        gene_lists.append(temp)
    return gene_lists


def add_gene_to_protein_chain(universal_graph, gene_id, label_name=None):
    """Adds gene → transcript → protein to graph, returns protein label."""
    if universal_graph.add_vertex(gene_id, label=[label_name or gene_id], vert_info={}, omic_type='gene'):
        return gene_id + '_protein'
    else:
        tx_label = gene_id + '_transcript'
        px_label = gene_id + '_protein'
        universal_graph.add_vertex(tx_label, label=[(label_name or gene_id) + '_transcript'], vert_info={}, omic_type='transcript')
        universal_graph.add_edge(gene_id, tx_label, int_info={
            'type': 'gene - transcript', 'interaction': {0: 'transcribed_to'}})
        universal_graph.add_vertex(px_label, label=[(label_name or gene_id) + '_protein'], vert_info={}, omic_type='protein')
        universal_graph.add_edge(tx_label, px_label, int_info={
            'type': 'transcript - protein', 'interaction': {0: 'translated_to'}})
        return px_label

def add_protein_complex(universal_graph, complex_id):
    """Adds a protein complex and its parts to the graph."""
    protein_list = complex_id.split(":")[1].split("_")
    universal_graph.add_vertex(complex_id, label=[complex_id], vert_info={'parts': protein_list}, omic_type='protein_complex')
    for pr in protein_list:
        pr_px = add_gene_to_protein_chain(universal_graph, pr, label_name=pr)
        universal_graph.add_edge(pr_px, complex_id, int_info={
            'type': 'protein - protein_complex', 'interaction': {0: 'part_of'}})

def add_interaction_edge(universal_graph, source, target, interaction, type_label=None, weight=None):
    """Adds an edge between two nodes."""
    int_info = {
        'type': type_label or f"{universal_graph.get_vertex(source).get_omic_type()} - {universal_graph.get_vertex(target).get_omic_type()}",
        'interaction': {0: interaction}
    }
    universal_graph.add_edge(source, target, weight=weight, int_info=int_info)


def print_graph_summary(source_name, universal_graph, unique_genes=set(), not_found_genes=set(), total_items=None,
                        empty_count=None):
    count_gene = sum(1 for v in universal_graph.get_vertices().values() if v.get_omic_type() == 'gene')

    print(f"\n--- {source_name} Summary ---")
    if total_items is not None:
        print(f"Total items in {source_name}: {total_items}")
    if empty_count is not None:
        print(f"Number of empty mappings in {source_name}: {empty_count}")

    print("Number of unique genes: ", len(unique_genes))
    print("Number of genes with missing UniProt ID: ", len(not_found_genes))
    print("Number of vertices: ", len(universal_graph.get_vertices()))
    print("Number of edges: ", len(universal_graph.get_edges()))
    print("Number of gene vertices: ", count_gene)

    omic_type_counts = defaultdict(int)
    for v in universal_graph.get_vertices().values():
        omic_type_counts[v.get_omic_type()] += 1

    for omic_type, count in omic_type_counts.items():
        print(f"{omic_type}: {count}")
    print(f"{source_name} processing finished.\n")

# Filtering Graph
def get_reverse_adjacency_list(graph):
    reverse_adj = defaultdict(list)
    for edge in graph.get_edges().values():
        src = edge.get_start_vertex().get_id()
        tgt = edge.get_end_vertex().get_id()
        reverse_adj[tgt].append(src)  # reverse the edge
    return reverse_adj

def get_nodes_that_can_reach_reactions(graph):
    reverse_adj = get_reverse_adjacency_list(graph)
    visited = set()

    # Collect all reaction node IDs
    reaction_nodes = [v.get_id() for v in graph.get_vertices().values() if v.get_omic_type() == 'R']

    # BFS from each reaction node in reversed graph
    queue = deque(reaction_nodes)
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        for neighbor in reverse_adj[current]:
            if neighbor not in visited:
                queue.append(neighbor)

    return visited  # nodes that can reach at least one reaction

if __name__ == '__main__':

    start = time.time()
    universal_graph = Graph()

    modelREcon3d = load_json_model('/home/cakmaklab/Documents/MetabOmics_Aycan/omicNetwork/Databases/Recon3D.json')
    all_genes = modelREcon3d.genes
    num_unique_genes = len(all_genes)

    print(f"Number of unique genes in Recon3D: {num_unique_genes}")

    empty = 0
    unique_gene = set()
    nogene = set()

    for r in modelREcon3d.reactions:
        if r.gene_reaction_rule != '':
            universal_graph.add_vertex(
                r.id,
                label=[r.id],
                vert_info={
                    'metabolites': [
                        {str(metabolite): coefficient for metabolite, coefficient in r.metabolites.items()}],
                    'gpr': parse_gpr_expression(str(r.gpr), recon3D_uniprot_mapping)
                },
                omic_type='R'
            )
            for gen in r.genes:
                unique_gene.add(gen.id)
                if gen.id in recon3D_uniprot_mapping:
                    uniprot_id = recon3D_uniprot_mapping[gen.id]
                    label = [uniprot_id]
                else:
                    nogene.add(gen.id)
                    label = []
                if label:
                    label_px = add_gene_to_protein_chain(universal_graph, label[0], label_name=gen.name)
                    add_interaction_edge(universal_graph, label_px, r.id, interaction='catalyzes')
        else:
            empty += 1

    # Summary for Recon3D
    print_graph_summary(source_name='Recon3D', universal_graph=universal_graph, unique_genes=unique_gene, not_found_genes=nogene, total_items=len(modelREcon3d.reactions),
                        empty_count=empty)

    # Adding TRRUST(TF-gene) database to network
    trrust_tfg = pd.read_csv('/home/cakmaklab/Documents/MetabOmics_Aycan/omicNetwork/Databases/trrust_uniprot_human.tsv', sep='\t')
    count = 1
    error = 0
    unique_gene_trrust = set()
    for j in trrust_tfg.values:
        label_tf = str(j[1])
        label_target = str(j[3])
        if j[4] == 'Unknown' :
            continue
        unique_gene_trrust.add(label_target)
        label_px_tf = add_gene_to_protein_chain(universal_graph, label_tf, label_name=j[0])
        label_px_target = add_gene_to_protein_chain(universal_graph, label_target, label_name=j[2])
        add_interaction_edge(universal_graph, label_px_tf, label_px_target, interaction=j[4])

    # Summary for TRRUST
    print_graph_summary(source_name='TRRUST', universal_graph=universal_graph, unique_genes=unique_gene_trrust,
                        total_items=len(trrust_tfg.values))

    # Adding mirtarbase(miRNA-gene) database to network
    mirTarBase = pd.read_csv('/home/cakmaklab/Documents/MetabOmics_Aycan/omicNetwork/Databases/mirTarBase_evidenceStrong.csv')
    count = 0
    error = 0
    unique_gene_mir = set()
    for j in mirTarBase.values:
        try:
            label_target = j[4]
            unique_gene_mir.add(label_target)
            label_px_target = add_gene_to_protein_chain(universal_graph, label_target, label_name=j[3])
            universal_graph.add_vertex(j[1], label=[j[1]], vert_info={'int_id': j[0]}, omic_type='miRNA')
            add_interaction_edge(universal_graph, j[1], label_target+'_transcript', interaction='Repression', weight= j[7])
        except Exception as e:
            error += 1
            universal_graph.remove_vertex(label_target)
            # print(f"Error processing {j}: {e}")
            continue

    print("error : ", error)
    # Summary for mirTarBase
    print_graph_summary(source_name='mirTarBase', universal_graph=universal_graph, unique_genes=unique_gene_mir,
                        total_items=len(mirTarBase.values))

    # Adding Omni-Path(protein-protein signaling) database to network
    omniPath = pd.read_csv('/home/cakmaklab/Documents/MetabOmics_Aycan/omicNetwork/Databases/filtered_omnipath_interactions.csv')
    complex_protein = set()
    unique_gene_omniPath = set()
    for row in omniPath.values:
        src, tgt = row[0], row[1]
        if row[3] == row[4]:
            continue
        interaction = 'Activation' if row[3] == 1 else 'Repression'

        is_src_complex = 'COMPLEX' in src
        is_tgt_complex = 'COMPLEX' in tgt
        if is_src_complex: add_protein_complex(universal_graph, src)
        if is_tgt_complex: add_protein_complex(universal_graph, tgt)

        if is_src_complex and is_tgt_complex :
            add_interaction_edge(universal_graph, src, tgt, interaction)
        elif is_src_complex :
            tgt_px = add_gene_to_protein_chain(universal_graph, tgt, label_name=row[0])
            add_interaction_edge(universal_graph, src, tgt_px, interaction)
        elif is_tgt_complex :
            src_px = add_gene_to_protein_chain(universal_graph, src, label_name=row[1])
            add_interaction_edge(universal_graph, src_px, tgt, interaction)
        else :
            src_px = add_gene_to_protein_chain(universal_graph, src, label_name=row[0])
            tgt_px = add_gene_to_protein_chain(universal_graph, tgt, label_name=row[1])
            add_interaction_edge(universal_graph, src_px, tgt_px, interaction)

        unique_gene_omniPath.update([src, tgt])

    # Summary for Omni-Path
    print_graph_summary(source_name='Omni-Path', universal_graph=universal_graph, unique_genes=unique_gene_omniPath,
                        total_items=len(omniPath.values))

    # Adding enhancer-promoter-gene interaction database to network from GeneHancer database
    # read prom/enh - gene interaction
    enh_prom_gene = pd.read_csv('/home/cakmaklab/Documents/MetabOmics_Aycan/omicNetwork/Databases/GeneHancer_AnnotSV_gene_association_scores_v5.25_filtered_proteinCoding.csv')
    # read TF - prom/enh interaction
    TF_enh_prom = pd.read_csv('/home/cakmaklab/Documents/MetabOmics_Aycan/omicNetwork/Databases/GeneHancer_TFBSs_v5.25_uniprotId.csv')
    # read prom/enh annotation
    enh_prom = pd.read_csv('/home/cakmaklab/Documents/MetabOmics_Aycan/omicNetwork/Databases/GeneHancer_AnnotSV_elements_v5.25_hg19_hg38_filtered.csv')

    unique_gene_enh_prom = set()
    # Add prom/enh - gene interaction to network
    merged_1 = enh_prom_gene.merge(
        enh_prom,
        on="GHid",
        how="inner"
    )
    print(merged_1.columns.tolist())
    for _, row in merged_1.iterrows():
        # add prom/enh vertex
        universal_graph.add_vertex(
            row["GHid"],
            label=[row["GHid"]],
            vert_info={"hg19": {'start' : row["start_hg19"], 'end':row["end_hg19"], 'chr' : row["chromosome_hg19"] },
                       "hg38": {'start' : row["element_start_hg38"], 'end':row["element_end_hg38"], 'chr' : row["chr_hg38"] },
                       "enhancer_score": row["enhancer_score_hg38"], "is_elite" : row["is_elite_hg38"]},
            omic_type=row['regulatory_element_type_hg38']
        )

        # add gene vertex
        target = row["uniprot_id"]
        unique_gene_enh_prom.add(target)
        target_px = add_gene_to_protein_chain(universal_graph, target, label_name=row["symbol"])
        add_interaction_edge(universal_graph, row["GHid"], target, interaction='Activation', weight=row["combined_score"])

    # Add prom/enh - TF interaction to network
    print(TF_enh_prom.columns.tolist())
    merged_2 = TF_enh_prom.merge(
        enh_prom,
        on="GHid",
        how="inner"
    )
    print(merged_2.columns.tolist())
    for _, row in merged_2.iterrows():
        # add prom/enh vertex
        universal_graph.add_vertex(
            row["GHid"],
            label=[row["GHid"]],
            vert_info={"hg19": {'start': row["start_hg19"], 'end': row["end_hg19"], 'chr': row["chromosome_hg19"]},
                       "hg38": {'start': row["element_start_hg38"], 'end': row["element_end_hg38"],
                                'chr': row["chr_hg38"]},
                       "enhancer_score": row["enhancer_score_hg38"], "is_elite": row["is_elite_hg38"]},
            omic_type=row['regulatory_element_type_hg38']
        )

        # add TF vertex
        src = row["uniprot_id"]
        unique_gene_enh_prom.add(src)
        src_px = add_gene_to_protein_chain(universal_graph, src, label_name=row["TF"])
        add_interaction_edge(universal_graph, src_px, row["GHid"], interaction='Activation')

    # Summary for geneHancer
    print_graph_summary(source_name='enh-prom-gene', universal_graph=universal_graph,
                        unique_genes=unique_gene_enh_prom,
                        total_items=len(enh_prom_gene.values))

    # Add CADD dataset, (SNVs - Genomic Variants) SNVs-promoter/gene interaction to network

    # read SNP_variants - gene interaction
    SNP_variants = pd.read_csv('/home/cakmaklab/Documents/MetabOmics_Aycan/omicNetwork/Databases/cadd_phred20_final_filtered.tsv', sep="\t")

    # Create key column
    SNP_variants['variant_key'] = SNP_variants['#Chrom'].astype(str) + '-' + SNP_variants['Pos'].astype(str) + '-' + SNP_variants['Ref'] + '-' + SNP_variants['Alt']
    ref_gen = "hg38"
    print("Promoter/Enhancer indeksi olusturuluyor (Bu islem RAM kullanacaktir)...", flush=True)
    genomic_index = {}
    ref_gen = "hg38" 

    
    for node in list(universal_graph.get_vertices()):
        vertex = universal_graph.get_vertex(node)
        if vertex.get_omic_type() in ['Promoter/Enhancer', 'Promoter']:
            vertex_info = vertex.get_vert_info()[ref_gen]
            
            chrom = str(vertex_info['chr']).replace('chr', '')
            start = int(vertex_info["start"])
            end = int(vertex_info["end"])
            
            if chrom not in genomic_index:
                genomic_index[chrom] = {}
                
            for pos in range(start, end + 1):
                if pos not in genomic_index[chrom]:
                    genomic_index[chrom][pos] = []
                genomic_index[chrom][pos].append(node)

    print("Indeksleme tamamlandi. CADD verisi islenmeye basliyor...", flush=True)


    count_snp = 0
    for j in SNP_variants.values:
        count_snp += 1
        if count_snp % 10000 == 0:
            print(f"Islemlenen SNP: {count_snp}", flush=True)
        universal_graph.add_vertex(j[9], label=[j[9]], vert_info={'PHRED': j[8]}, omic_type='snp')
        
        '''if j[6] in synonym_mapping_dict :
            uniprot_id = synonym_mapping_dict[j[6]]
            linked_vertex = next((uid for uid in uniprot_id if uid[0] in ['P', 'Q', 'O']), uniprot_id[0])
            add_interaction_edge(universal_graph, j[9], linked_vertex, interaction='snp_variation', weight=j[8])
        elif j[7] in synonym_mapping_dict :
            uniprot_id = synonym_mapping_dict[j[7]]
            linked_vertex = next((uid for uid in uniprot_id if uid[0] in ['P', 'Q', 'O']), uniprot_id[0])
            add_interaction_edge(universal_graph, j[9], linked_vertex, interaction='snp_variation', weight=j[8])
        else :
            for node in list(universal_graph.get_vertices()):
                if universal_graph.get_vertex(node).get_omic_type() in ['Promoter/Enhancer', 'Promoter']:
                    vertex_info = universal_graph.get_vertex(node).get_vert_info()[ref_gen]
                    #chr_id, start, end = vertex_info.split("-")
                    start = int(vertex_info["start"])
                    end = int(vertex_info["end"])
                    if j[0] != vertex_info['chr'].replace('chr', ''):
                        continue
                    if j[1] >= vertex_info['start'] and j[1] <= vertex_info['end']:
                        add_interaction_edge(universal_graph, j[9], node, interaction='snp_variation', weight=j[8])'''
        # Try j[6] first, then j[7]
        for key in [j[6], j[7]]:
            if key in synonym_mapping_dict:
                uniprot_id = synonym_mapping_dict[key]

                # Step 1: Pick P/Q/O IDs in vertices
                linked_vertex = next(
                    (uid for uid in uniprot_id if uid[0] in ['P', 'Q', 'O'] and uid in universal_graph.get_vertices()),
                    None
                )

                # Step 2: If none, pick any ID in vertices
                if linked_vertex is None:
                    linked_vertex = next(
                        (uid for uid in uniprot_id if uid in universal_graph.get_vertices()),
                        None
                    )

                # Step 3: Add edge only if linked_vertex exists
                if linked_vertex is not None:
                    add_interaction_edge(
                        universal_graph,
                        j[9],
                        linked_vertex,
                        interaction='snp_variation',
                        weight=j[8]
                    )
                    break  # Stop after first valid linked_vertex

        else:
            # # If neither j[6] nor j[7] produced a vertex, fallback to promoter/enhancer search
            # for node in list(universal_graph.get_vertices()):
            #     vertex = universal_graph.get_vertex(node)
            #     if vertex.get_omic_type() in ['Promoter/Enhancer', 'Promoter']:
            #         vertex_info = vertex.get_vert_info()[ref_gen]
            #         start = int(vertex_info["start"])
            #         end = int(vertex_info["end"])

            #         # Skip if chromosome doesn't match
            #         if j[0] != vertex_info['chr'].replace('chr', ''):
            #             continue

            #         # Check if SNP falls within the vertex region
            #         if j[1] >= start and j[1] <= end:
            #             add_interaction_edge(
            #                 universal_graph,
            #                 j[9],
            #                 node,
            #                 interaction='snp_variation',
            #                 weight=j[8]
            #             )
            # ---------------------------------------------------------
            snp_chrom = str(j[0])
            snp_pos = int(j[1])

            if snp_chrom in genomic_index and snp_pos in genomic_index[snp_chrom]:
                matched_nodes = genomic_index[snp_chrom][snp_pos]
                
                for node in matched_nodes:
                    add_interaction_edge(
                        universal_graph,
                        j[9],
                        node,
                        interaction='snp_variation',
                        weight=j[8]
                    )
            # ---------------------------------------------------------
    # Summary for CADD
    print_graph_summary(source_name='SNP_variants-gene', universal_graph=universal_graph,
                        #unique_genes=unique_gene_enh_prom,
                        total_items=len(SNP_variants.values))

    universal_graph.save_to_json("/home/cakmaklab/Documents/MetabOmics_Aycan/omicNetwork/Databases/universalGraph_with_omniPath_enh_prom_SNP_unFiltered.json")

    # Filter out nodes that cannot reach any reaction node
    print("BFS Filtreleme islemi basladi", flush=True)
    reachable_nodes = get_nodes_that_can_reach_reactions(universal_graph)
    print("BFS Filtreleme islemi bitti.", flush=True)
    all_nodes = set(universal_graph.get_vertices().keys())
    unreachable_nodes = all_nodes - reachable_nodes

    for node_id in unreachable_nodes:
        try:
            universal_graph.remove_vertex(node_id)
        except ValueError:
            pass  # vertex might already have been removed during earlier clean-up
    #""
    print(f"Filtered out {len(unreachable_nodes)} unreachable nodes.")
    print(f"Remaining vertices: {len(universal_graph.get_vertices())}")
    print(f"Remaining edges: {len(universal_graph.get_edges())}")

    print_graph_summary(source_name='Omni-Path', universal_graph=universal_graph, unique_genes=unique_gene_omniPath,
                        total_items=len(omniPath.values))

    universal_graph.save_to_json("/home/cakmaklab/Documents/MetabOmics_Aycan/omicNetwork/Databases/universalGraph_with_omniPath_enh_prom_SNP_Filtered.json")
    end = time.time()
    print(f"Elapsed time : ", round((end - start) / 60.0, 2), " minutes")

    # write network information to the file
    info = "Number of vertex : " + str(len(universal_graph.get_vertices())) + "\nNumber of edge : " + str(
        len(universal_graph.get_edges())) + f"\nElapsed time : " + str(round((end - start) / 60.0, 2)) + " minutes"
    # json_file_path = 'toyExample/toyDataset/toy_networkInfo.txt'
    json_file_path = '/home/cakmaklab/Documents/MetabOmics_Aycan/omicNetwork/Databases/universalGraph_with_omniPath_enh_promFiltered.txt'
    f = open(json_file_path, "w")
    f.write(info)
